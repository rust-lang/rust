#[link(name="socketlib", vers="0.0")];
#[crate_type = "lib"];
#[legacy_exports];

mod socket {
    #[legacy_exports];

export socket_handle;

struct socket_handle {
    sockfd: libc::c_int,
}

impl socket_handle : Drop {
    fn finalize() {
        /* c::close(self.sockfd); */
    }
}

    fn socket_handle(x: libc::c_int) -> socket_handle {
        socket_handle {
            sockfd: x
        }
    }

}
