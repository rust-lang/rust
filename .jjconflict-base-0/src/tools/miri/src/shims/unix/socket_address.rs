use std::iter;
use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV4, SocketAddrV6, ToSocketAddrs};

use rustc_abi::Size;
use rustc_target::spec::Env;

use crate::*;

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn getaddrinfo(
        &mut self,
        node: &OpTy<'tcx>,
        service: &OpTy<'tcx>,
        hints: &OpTy<'tcx>,
        res: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let node_ptr = this.read_pointer(node)?;
        let service_ptr = this.read_pointer(service)?;
        let hints_ptr = this.read_pointer(hints)?;
        let res_mplace = this.deref_pointer(res)?;

        if node_ptr == Pointer::null() {
            // We cannot get an address without the `node` part because
            // the [`ToSocketAddrs`] trait requires an address.
            throw_unsup_format!(
                "getaddrinfo: getting the address info without a `node` is unsupported"
            );
        }

        let mut port = 0;
        if service_ptr != Pointer::null() {
            // The C-string at `service_ptr` is either a port number or a name of a
            // well-known service.
            let service_c_str = this.read_c_str(service_ptr)?;

            // Try to parse `service_c_str` as a number -- the only case we support.
            match str::from_utf8(service_c_str).ok().and_then(|s| s.parse::<u16>().ok()) {
                Some(service_port) => port = service_port,
                None => {
                    // The string is not a valid port number; this is unsupported
                    // because the standard library's [`ToSocketAddrs`] only supports
                    // numeric ports.
                    throw_unsup_format!(
                        "getaddrinfo: non-numeric `service` arguments aren't supported"
                    )
                }
            }
        }

        let node_c_str = this.read_c_str(node_ptr)?;
        let Some(node_str) = str::from_utf8(node_c_str).ok() else {
            throw_unsup_format!("getaddrinfo: node is not a valid UTF-8 string")
        };

        if hints_ptr == Pointer::null() {
            // The standard library only supports getting TCP address information. The
            // empty hints pointer would allow any socket type so we cannot support it.
            throw_unsup_format!(
                "getaddrinfo: getting address info without providing socket type hint is unsupported"
            )
        }

        let hints_layout = this.libc_ty_layout("addrinfo");
        let hints_mplace = this.ptr_to_mplace(hints_ptr, hints_layout);

        let family_field = this.project_field_named(&hints_mplace, "ai_family")?;
        let family = this.read_scalar(&family_field)?;
        if family != Scalar::from_i32(0) {
            // We cannot provide a family hint to the standard library implementation.
            throw_unsup_format!("getaddrinfo: family hints are not supported")
        }

        let socktype_field = this.project_field_named(&hints_mplace, "ai_socktype")?;
        let socktype = this.read_scalar(&socktype_field)?;
        if socktype != this.eval_libc("SOCK_STREAM") {
            // The standard library only supports getting TCP address information.
            throw_unsup_format!(
                "getaddrinfo: only queries with socket type SOCK_STREAM are supported"
            )
        }

        let protocol_field = this.project_field_named(&hints_mplace, "ai_protocol")?;
        let protocol = this.read_scalar(&protocol_field)?;
        if protocol != Scalar::from_i32(0) {
            // We cannot provide a protocol hint to the standard library implementation.
            throw_unsup_format!("getaddrinfo: protocol hints are not supported")
        }

        let flags_field = this.project_field_named(&hints_mplace, "ai_flags")?;
        let flags = this.read_scalar(&flags_field)?;
        if flags != Scalar::from_i32(0) {
            // We cannot provide any flag hints to the standard library implementation.
            throw_unsup_format!("getaddrinfo: flag hints are not supported")
        }

        let socket_addrs = match (node_str, port).to_socket_addrs() {
            Ok(addrs) => addrs,
            Err(e) => {
                // `getaddrinfo` returns negative integer values when there was an error during socket
                // address resolution. Because the standard library doesn't expose those integer values
                // directly, we just return a generic protocol error.
                // The actual error is emitted as part of a warning diagnostic.
                this.emit_diagnostic(NonHaltingDiagnostic::SocketAddressResolution { error: e });
                this.set_last_error(LibcError("EPROTO"))?;
                return interp_ok(this.eval_libc("EAI_SYSTEM"));
            }
        };

        let res_ptr = this.allocate_address_infos(socket_addrs)?;

        this.write_pointer(res_ptr, &res_mplace)?;
        interp_ok(Scalar::from_i32(0))
    }

    fn freeaddrinfo(&mut self, res: &OpTy<'tcx>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let res_ptr = this.read_pointer(res)?;

        this.free_address_infos(res_ptr)
    }

    /// Attempt to turn an address and length operand into a standard library socket address.
    ///
    /// Returns an IO error should the address length not match the address family length.
    fn read_socket_address(
        &self,
        address: &OpTy<'tcx>,
        address_len: &OpTy<'tcx>,
        foreign_name: &'static str,
    ) -> InterpResult<'tcx, Result<SocketAddr, IoError>> {
        let this = self.eval_context_ref();

        let socklen_layout = this.libc_ty_layout("socklen_t");
        // We only support address lengths which can be stored in a u64 since the
        // size of a layout is also stored in a u64.
        let address_len: u64 =
            this.read_scalar(address_len)?.to_int(socklen_layout.size)?.try_into().unwrap();

        // Initially, treat address as generic sockaddr just to extract the family field.
        let sockaddr_layout = this.libc_ty_layout("sockaddr");
        if address_len < sockaddr_layout.size.bytes() {
            // Address length should be at least as big as the generic sockaddr
            return interp_ok(Err(LibcError("EINVAL")));
        }
        let address = this.deref_pointer_as(address, sockaddr_layout)?;

        let family_field = this.project_field_named(&address, "sa_family")?;
        let family_layout = this.libc_ty_layout("sa_family_t");
        let family = this.read_scalar(&family_field)?.to_int(family_layout.size)?;

        // Depending on the family, decide whether it's IPv4 or IPv6 and use specialized layout
        // to extract address and port.
        let socket_addr = if family == this.eval_libc_i32("AF_INET").into() {
            let sockaddr_in_layout = this.libc_ty_layout("sockaddr_in");
            if address_len != sockaddr_in_layout.size.bytes() {
                // Address length should be exactly the length of an IPv4 address.
                return interp_ok(Err(LibcError("EINVAL")));
            }
            let address = address.transmute(sockaddr_in_layout, this)?;

            let port_field = this.project_field_named(&address, "sin_port")?;
            // Read bytes and treat them as big endian since port is stored in network byte order.
            let port_bytes: [u8; 2] = this
                .read_bytes_ptr_strip_provenance(port_field.ptr(), Size::from_bytes(2))?
                .try_into()
                .unwrap();
            let port = u16::from_be_bytes(port_bytes);

            let addr_field = this.project_field_named(&address, "sin_addr")?;
            let s_addr_field = this.project_field_named(&addr_field, "s_addr")?;
            // Read bytes and treat them as big endian since address is stored in network byte order.
            let addr_bytes: [u8; 4] = this
                .read_bytes_ptr_strip_provenance(s_addr_field.ptr(), Size::from_bytes(4))?
                .try_into()
                .unwrap();
            let addr_bits = u32::from_be_bytes(addr_bytes);

            SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::from_bits(addr_bits), port))
        } else if family == this.eval_libc_i32("AF_INET6").into() {
            let sockaddr_in6_layout = this.libc_ty_layout("sockaddr_in6");
            if address_len != sockaddr_in6_layout.size.bytes() {
                // Address length should be exactly the length of an IPv6 address.
                return interp_ok(Err(LibcError("EINVAL")));
            }
            // We cannot transmute since the `sockaddr_in6` layout is bigger than the `sockaddr` layout.
            let address = address.offset(Size::ZERO, sockaddr_in6_layout, this)?;

            let port_field = this.project_field_named(&address, "sin6_port")?;
            // Read bytes and treat them as big endian since port is stored in network byte order.
            let port_bytes: [u8; 2] = this
                .read_bytes_ptr_strip_provenance(port_field.ptr(), Size::from_bytes(2))?
                .try_into()
                .unwrap();
            let port = u16::from_be_bytes(port_bytes);

            let addr_field = this.project_field_named(&address, "sin6_addr")?;
            let s_addr_field = this
                .project_field_named(&addr_field, "s6_addr")?
                .transmute(this.machine.layouts.u128, this)?;
            // Read bytes and treat them as big endian since address is stored in network byte order.
            let addr_bytes: [u8; 16] = this
                .read_bytes_ptr_strip_provenance(s_addr_field.ptr(), Size::from_bytes(16))?
                .try_into()
                .unwrap();
            let addr_bits = u128::from_be_bytes(addr_bytes);

            let flowinfo_field = this.project_field_named(&address, "sin6_flowinfo")?;
            // flowinfo doesn't get the big endian treatment as this field is stored in native byte order
            // and not in network byte order.
            let flowinfo = this.read_scalar(&flowinfo_field)?.to_u32()?;

            let scope_id_field = this.project_field_named(&address, "sin6_scope_id")?;
            // scope_id doesn't get the big endian treatment as this field is stored in native byte order
            // and not in network byte order.
            let scope_id = this.read_scalar(&scope_id_field)?.to_u32()?;

            SocketAddr::V6(SocketAddrV6::new(
                Ipv6Addr::from_bits(addr_bits),
                port,
                flowinfo,
                scope_id,
            ))
        } else {
            // Socket of other types shouldn't be created in a first place and
            // thus also no address family of another type should be supported.
            throw_unsup_format!(
                "{foreign_name}: address family {family:#x} is unsupported, \
                only AF_INET and AF_INET6 are allowed"
            );
        };

        interp_ok(Ok(socket_addr))
    }

    /// Attempt to write a standard library socket address into a pointer.
    ///
    /// The `address_len_ptr` parameter serves both as input and output parameter.
    /// On input, it points to the size of the buffer `address_ptr` points to, and
    /// on output it points to the non-truncated size of the written address in the
    /// buffer pointed to by `address_ptr`.
    ///
    /// If the address buffer doesn't fit the whole address, the address is truncated to not
    /// overflow the buffer.
    fn write_socket_address(
        &mut self,
        address: &SocketAddr,
        address_ptr: Pointer,
        address_len_ptr: Pointer,
        foreign_name: &'static str,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        if address_ptr == Pointer::null() || address_len_ptr == Pointer::null() {
            // The POSIX man page doesn't account for the cases where the `address_ptr` or
            // `address_len_ptr` could be null pointers. Thus, this behavior is undefined!
            throw_ub_format!(
                "{foreign_name}: writing a socket address but the address or the length pointer is a null pointer"
            )
        }

        let socklen_layout = this.libc_ty_layout("socklen_t");
        let address_buffer_len_place = this.ptr_to_mplace(address_len_ptr, socklen_layout);
        // We only support buffer lengths which can be stored in a u64 since the
        // size of a layout in bytes is also stored in a u64.
        let address_buffer_len: u64 = this
            .read_scalar(&address_buffer_len_place)?
            .to_int(socklen_layout.size)?
            .try_into()
            .unwrap();

        let address_buffer = match address {
            SocketAddr::V4(address) => {
                // IPv4 address bytes; already stored in network byte order.
                let address_bytes = address.ip().octets();
                // Port needs to be manually turned into network byte order.
                let port = address.port().to_be();

                let sockaddr_in_layout = this.libc_ty_layout("sockaddr_in");
                // Allocate new buffer on the stack with the `sockaddr_in` layout.
                // We need a temporary buffer as `address_ptr` might not point to a large enough
                // buffer, in which case we have to truncate.
                let address_buffer = this.allocate(sockaddr_in_layout, MemoryKind::Stack)?;
                // Zero the whole buffer as some libc targets have additional fields which we fill
                // with zero bytes (just like the standard library does it).
                this.write_bytes_ptr(
                    address_buffer.ptr(),
                    iter::repeat_n(0, address_buffer.layout.size.bytes_usize()),
                )?;

                let sin_family_field = this.project_field_named(&address_buffer, "sin_family")?;
                // We cannot simply write the `AF_INET` scalar into the `sin_family_field` because on most
                // systems the field has a layout of 16-bit whilst the scalar has a size of 32-bit.
                // Since the `AF_INET` constant is chosen such that it can safely be converted into
                // a 16-bit integer, we use the following logic to get a scalar of the right size.
                let af_inet = this.eval_libc("AF_INET");
                let address_family =
                    Scalar::from_int(af_inet.to_int(af_inet.size())?, sin_family_field.layout.size);
                this.write_scalar(address_family, &sin_family_field)?;

                let sin_port_field = this.project_field_named(&address_buffer, "sin_port")?;
                // Write the port in target native endianness bytes as we already converted it
                // to big endian above.
                this.write_bytes_ptr(sin_port_field.ptr(), port.to_ne_bytes())?;

                let sin_addr_field = this.project_field_named(&address_buffer, "sin_addr")?;
                let s_addr_field = this.project_field_named(&sin_addr_field, "s_addr")?;
                this.write_bytes_ptr(s_addr_field.ptr(), address_bytes)?;

                address_buffer
            }
            SocketAddr::V6(address) => {
                // IPv6 address bytes; already stored in network byte order.
                let address_bytes = address.ip().octets();
                // Port needs to be manually turned into network byte order.
                let port = address.port().to_be();
                // Flowinfo is stored in native byte order.
                let flowinfo = address.flowinfo();
                // Scope id is stored in native byte order.
                let scope_id = address.scope_id();

                let sockaddr_in6_layout = this.libc_ty_layout("sockaddr_in6");
                // Allocate new buffer on the stack with the `sockaddr_in6` layout.
                // We need a temporary buffer as `address_ptr` might not point to a large enough
                // buffer, in which case we have to truncate.
                let address_buffer = this.allocate(sockaddr_in6_layout, MemoryKind::Stack)?;
                // Zero the whole buffer as some libc targets have additional fields which we fill
                // with zero bytes (just like the standard library does it).
                this.write_bytes_ptr(
                    address_buffer.ptr(),
                    iter::repeat_n(0, address_buffer.layout.size.bytes_usize()),
                )?;

                let sin6_family_field = this.project_field_named(&address_buffer, "sin6_family")?;
                // We cannot simply write the `AF_INET6` scalar into the `sin6_family_field` because on most
                // systems the field has a layout of 16-bit whilst the scalar has a size of 32-bit.
                // Since the `AF_INET6` constant is chosen such that it can safely be converted into
                // a 16-bit integer, we use the following logic to get a scalar of the right size.
                let af_inet6 = this.eval_libc("AF_INET6");
                let address_family = Scalar::from_int(
                    af_inet6.to_int(af_inet6.size())?,
                    sin6_family_field.layout.size,
                );
                this.write_scalar(address_family, &sin6_family_field)?;

                let sin6_port_field = this.project_field_named(&address_buffer, "sin6_port")?;
                // Write the port in target native endianness bytes as we already converted it
                // to big endian above.
                this.write_bytes_ptr(sin6_port_field.ptr(), port.to_ne_bytes())?;

                let sin6_flowinfo_field =
                    this.project_field_named(&address_buffer, "sin6_flowinfo")?;
                this.write_scalar(Scalar::from_u32(flowinfo), &sin6_flowinfo_field)?;

                let sin6_scope_id_field =
                    this.project_field_named(&address_buffer, "sin6_scope_id")?;
                this.write_scalar(Scalar::from_u32(scope_id), &sin6_scope_id_field)?;

                let sin6_addr_field = this.project_field_named(&address_buffer, "sin6_addr")?;
                let s6_addr_field = this.project_field_named(&sin6_addr_field, "s6_addr")?;
                this.write_bytes_ptr(s6_addr_field.ptr(), address_bytes)?;

                address_buffer
            }
        };

        // Copy the truncated address into the pointer pointed to by `address_ptr`.
        this.mem_copy(
            address_buffer.ptr(),
            address_ptr,
            // Truncate the address to fit the provided buffer.
            address_buffer.layout.size.min(Size::from_bytes(address_buffer_len)),
            // The buffers are guaranteed to not overlap since the `address_buffer`
            // was just newly allocated on the stack.
            true,
        )?;
        // Deallocate the address buffer as it was only needed to construct the address and
        // copy it into the buffer pointed to by `address_ptr`.
        this.deallocate_ptr(address_buffer.ptr(), None, MemoryKind::Stack)?;
        // Size of the non-truncated address.
        let address_len = address_buffer.layout.size.bytes();

        this.write_scalar(
            Scalar::from_uint(address_len, socklen_layout.size),
            &address_buffer_len_place,
        )?;

        interp_ok(())
    }
}

impl<'tcx> EvalContextPrivExt<'tcx> for crate::MiriInterpCx<'tcx> {}
trait EvalContextPrivExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Allocate a linked list of address info structs from an iterator of [`SocketAddr`]s.
    /// Returns a pointer pointing to the head of the linked list.
    fn allocate_address_infos(
        &mut self,
        mut addresses: impl Iterator<Item = SocketAddr>,
    ) -> InterpResult<'tcx, Pointer> {
        let this = self.eval_context_mut();

        let Some(address) = addresses.next() else {
            // Iterator is empty; we return a null pointer.
            return interp_ok(Pointer::null());
        };

        let addrinfo_layout = this.libc_ty_layout("addrinfo");

        let addrinfo_mplace =
            this.allocate(addrinfo_layout, MiriMemoryKind::SocketAddress.into())?;

        let flags_mplace = this.project_field_named(&addrinfo_mplace, "ai_flags")?;
        // We don't support flag hints and depending on the target libc we have different default values:
        // "According to POSIX.1, specifying hints as NULL should cause `ai_flags` to be assumed as 0.
        // The GNU C library instead assumes a value of (AI_V4MAPPED | AI_ADDRCONFIG) for this case,
        // since this value is considered an improvement on the specification."
        let flags = if matches!(this.tcx.sess.target.env, Env::Gnu) {
            this.eval_libc_i32("AI_V4MAPPED") | this.eval_libc_i32("AI_ADDRCONFIG")
        } else {
            0
        };
        this.write_int(flags, &flags_mplace)?;

        let family_mplace = this.project_field_named(&addrinfo_mplace, "ai_family")?;
        let family = match &address {
            SocketAddr::V4(_) => this.eval_libc("AF_INET"),
            SocketAddr::V6(_) => this.eval_libc("AF_INET6"),
        };
        this.write_scalar(family, &family_mplace)?;

        let socktype_mplace = this.project_field_named(&addrinfo_mplace, "ai_socktype")?;
        this.write_scalar(this.eval_libc("SOCK_STREAM"), &socktype_mplace)?;

        let protocol_mplace = this.project_field_named(&addrinfo_mplace, "ai_protocol")?;
        // We don't support protocol hints and thus we just return zero which falls back
        // to the default protocol for the provided socket type.
        this.write_int(0, &protocol_mplace)?;

        // `sockaddr_storage` is guaranteed to fit any `sockaddr_*` address structure.
        let sockaddr_layout = this.libc_ty_layout("sockaddr_storage");

        let addrlen_mplace = this.project_field_named(&addrinfo_mplace, "ai_addrlen")?;
        let addr_mplace = this.project_field_named(&addrinfo_mplace, "ai_addr")?;
        this.write_int(sockaddr_layout.size.bytes(), &addrlen_mplace)?;

        let sockaddr_mplace = this.allocate(sockaddr_layout, MiriMemoryKind::Machine.into())?;
        // Zero the newly allocated socket address struct.
        this.write_bytes_ptr(
            sockaddr_mplace.ptr(),
            iter::repeat_n(0, sockaddr_mplace.layout.size.bytes_usize()),
        )?;
        this.write_socket_address(
            &address,
            sockaddr_mplace.ptr(),
            addrlen_mplace.ptr(),
            "getaddrinfo",
        )?;
        this.write_pointer(sockaddr_mplace.ptr(), &addr_mplace)?;

        let canonname_mplace = this.project_field_named(&addrinfo_mplace, "ai_canonname")?;
        this.write_pointer(Pointer::null(), &canonname_mplace)?;

        // Allocate remaining list and store a pointer to it.
        let next_mplace = this.project_field_named(&addrinfo_mplace, "ai_next")?;
        let next_ptr = this.allocate_address_infos(addresses)?;
        this.write_pointer(next_ptr, &next_mplace)?;

        interp_ok(addrinfo_mplace.ptr())
    }

    /// Deallocate the linked list of address info structs.
    /// `address_ptr` points to the start from where we deallocate recursively.
    fn free_address_infos(&mut self, address_ptr: Pointer) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        if address_ptr == Pointer::null() {
            // We're at the end of the linked list.
            return interp_ok(());
        }

        let addrinfo_layout = this.libc_ty_layout("addrinfo");
        let addrinfo_mplace = this.ptr_to_mplace(address_ptr, addrinfo_layout);

        let addr_field = this.project_field_named(&addrinfo_mplace, "ai_addr")?;
        let addr_ptr = this.read_pointer(&addr_field)?;
        this.deallocate_ptr(addr_ptr, None, MiriMemoryKind::Machine.into())?;

        let next_field = this.project_field_named(&addrinfo_mplace, "ai_next")?;
        let next_ptr = this.read_pointer(&next_field)?;
        this.free_address_infos(next_ptr)?;

        this.deallocate_ptr(address_ptr, None, MiriMemoryKind::SocketAddress.into())?;

        interp_ok(())
    }
}
