#!/usr/bin/env python3

# Ad-hoc script to print BTF file in a readable format.
# Follows the same printing conventions as bpftool with format 'raw'.
# Usage:
#
#   ./print_btf.py <btf_file>
#
# Parameters:
#
#   <btf_file> :: a file name or '-' to read from stdin.
#
# Intended usage:
#
#   llvm-objcopy --dump-section .BTF=- <input> | ./print_btf.py -
#
# Kernel documentation contains detailed format description:
#   https://www.kernel.org/doc/html/latest/bpf/btf.html

import struct
import ctypes
import sys


class SafeDict(dict):
    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            return f"<BAD_KEY: {key}>"


KINDS = SafeDict(
    {
        0: "UNKN",
        1: "INT",
        2: "PTR",
        3: "ARRAY",
        4: "STRUCT",
        5: "UNION",
        6: "ENUM",
        7: "FWD",
        8: "TYPEDEF",
        9: "VOLATILE",
        10: "CONST",
        11: "RESTRICT",
        12: "FUNC",
        13: "FUNC_PROTO",
        14: "VAR",
        15: "DATASEC",
        16: "FLOAT",
        17: "DECL_TAG",
        18: "TYPE_TAG",
        19: "ENUM64",
    }
)

INT_ENCODING = SafeDict(
    {0 << 0: "(none)", 1 << 0: "SIGNED", 1 << 1: "CHAR", 1 << 2: "BOOL"}
)

ENUM_ENCODING = SafeDict({0: "UNSIGNED", 1: "SIGNED"})

FUNC_LINKAGE = SafeDict({0: "static", 1: "global", 2: "extern"})

VAR_LINKAGE = SafeDict({0: "static", 1: "global", 2: "extern"})

FWD_KIND = SafeDict(
    {
        0: "struct",
        1: "union",
    }
)

for val, name in KINDS.items():
    globals()["BTF_KIND_" + name] = val


def warn(message):
    print(message, file=sys.stderr)


def print_btf(filename):
    if filename == "-":
        buf = sys.stdin.buffer.read()
    else:
        with open(filename, "rb") as file:
            buf = file.read()

    fmt_cache = {}
    endian_pfx = ">"  # big endian
    off = 0

    def unpack(fmt):
        nonlocal off, endian_pfx
        fmt = endian_pfx + fmt
        if fmt not in fmt_cache:
            fmt_cache[fmt] = struct.Struct(fmt)
        st = fmt_cache[fmt]
        r = st.unpack_from(buf, off)
        off += st.size
        return r

    # Use magic number at the header start to determine endianness
    (magic,) = unpack("H")
    if magic == 0xEB9F:
        endian_pfx = ">"  # big endian
    elif magic == 0x9FEB:
        endian_pfx = "<"  # little endian
    else:
        warn(f"Unexpected BTF magic: {magic:02x}")
        return

    # Rest of the header
    version, flags, hdr_len = unpack("BBI")
    type_off, type_len, str_off, str_len = unpack("IIII")

    # Offsets in the header are relative to the end of a header
    type_off += hdr_len
    str_off += hdr_len
    off = hdr_len
    type_end = type_off + type_len

    def string(rel_off):
        try:
            start = str_off + rel_off
            end = buf.index(b"\0", start)
            if start == end:
                return "(anon)"
            return buf[start:end].decode("utf8")
        except ValueError as e:
            warn(f"Can't get string at offset {str_off} + {rel_off}: {e}")
            return f"<BAD_STRING {rel_off}>"

    idx = 1
    while off < type_end:
        name_off, info, size = unpack("III")
        kind = (info >> 24) & 0x1F
        vlen = info & 0xFFFF
        kflag = info >> 31
        kind_name = KINDS[kind]
        name = string(name_off)

        def warn_nonzero(val, name):
            nonlocal idx
            if val != 0:
                warn(f"<{idx}> {name} should be 0 but is {val}")

        if kind == BTF_KIND_INT:
            (info,) = unpack("I")
            encoding = (info & 0x0F000000) >> 24
            offset = (info & 0x00FF0000) >> 16
            bits = info & 0x000000FF
            enc_name = INT_ENCODING[encoding]
            print(
                f"[{idx}] {kind_name} '{name}' size={size} "
                f"bits_offset={offset} "
                f"nr_bits={bits} encoding={enc_name}"
            )
            warn_nonzero(kflag, "kflag")
            warn_nonzero(vlen, "vlen")

        elif kind in [
            BTF_KIND_PTR,
            BTF_KIND_CONST,
            BTF_KIND_VOLATILE,
            BTF_KIND_RESTRICT,
        ]:
            print(f"[{idx}] {kind_name} '{name}' type_id={size}")
            warn_nonzero(name_off, "name_off")
            warn_nonzero(kflag, "kflag")
            warn_nonzero(vlen, "vlen")

        elif kind == BTF_KIND_ARRAY:
            warn_nonzero(name_off, "name_off")
            warn_nonzero(kflag, "kflag")
            warn_nonzero(vlen, "vlen")
            warn_nonzero(size, "size")
            type, index_type, nelems = unpack("III")
            print(
                f"[{idx}] {kind_name} '{name}' type_id={type} "
                f"index_type_id={index_type} nr_elems={nelems}"
            )

        elif kind in [BTF_KIND_STRUCT, BTF_KIND_UNION]:
            print(f"[{idx}] {kind_name} '{name}' size={size} vlen={vlen}")
            if kflag not in [0, 1]:
                warn(f"<{idx}> kflag should 0 or 1: {kflag}")
            for _ in range(0, vlen):
                name_off, type, offset = unpack("III")
                if kflag == 0:
                    print(
                        f"\t'{string(name_off)}' type_id={type} "
                        f"bits_offset={offset}"
                    )
                else:
                    bits_offset = offset & 0xFFFFFF
                    bitfield_size = offset >> 24
                    print(
                        f"\t'{string(name_off)}' type_id={type} "
                        f"bits_offset={bits_offset} "
                        f"bitfield_size={bitfield_size}"
                    )

        elif kind == BTF_KIND_ENUM:
            encoding = ENUM_ENCODING[kflag]
            print(
                f"[{idx}] {kind_name} '{name}' encoding={encoding} "
                f"size={size} vlen={vlen}"
            )
            for _ in range(0, vlen):
                (name_off,) = unpack("I")
                (val,) = unpack("i" if kflag == 1 else "I")
                print(f"\t'{string(name_off)}' val={val}")

        elif kind == BTF_KIND_ENUM64:
            encoding = ENUM_ENCODING[kflag]
            print(
                f"[{idx}] {kind_name} '{name}' encoding={encoding} "
                f"size={size} vlen={vlen}"
            )
            for _ in range(0, vlen):
                name_off, lo, hi = unpack("III")
                val = hi << 32 | lo
                if kflag == 1:
                    val = ctypes.c_long(val).value
                print(f"\t'{string(name_off)}' val={val}LL")

        elif kind == BTF_KIND_FWD:
            print(f"[{idx}] {kind_name} '{name}' fwd_kind={FWD_KIND[kflag]}")
            warn_nonzero(vlen, "vlen")
            warn_nonzero(size, "size")

        elif kind in [BTF_KIND_TYPEDEF, BTF_KIND_TYPE_TAG]:
            print(f"[{idx}] {kind_name} '{name}' type_id={size}")
            warn_nonzero(kflag, "kflag")
            warn_nonzero(kflag, "vlen")

        elif kind == BTF_KIND_FUNC:
            linkage = FUNC_LINKAGE[vlen]
            print(f"[{idx}] {kind_name} '{name}' type_id={size} " f"linkage={linkage}")
            warn_nonzero(kflag, "kflag")

        elif kind == BTF_KIND_FUNC_PROTO:
            print(f"[{idx}] {kind_name} '{name}' ret_type_id={size} " f"vlen={vlen}")
            warn_nonzero(name_off, "name_off")
            warn_nonzero(kflag, "kflag")
            for _ in range(0, vlen):
                name_off, type = unpack("II")
                print(f"\t'{string(name_off)}' type_id={type}")

        elif kind == BTF_KIND_VAR:
            (linkage,) = unpack("I")
            linkage = VAR_LINKAGE[linkage]
            print(f"[{idx}] {kind_name} '{name}' type_id={size}, " f"linkage={linkage}")
            warn_nonzero(kflag, "kflag")
            warn_nonzero(vlen, "vlen")

        elif kind == BTF_KIND_DATASEC:
            print(f"[{idx}] {kind_name} '{name}' size={size} vlen={vlen}")
            warn_nonzero(kflag, "kflag")
            warn_nonzero(size, "size")
            for _ in range(0, vlen):
                type, offset, size = unpack("III")
                print(f"\ttype_id={type} offset={offset} size={size}")

        elif kind == BTF_KIND_FLOAT:
            print(f"[{idx}] {kind_name} '{name}' size={size}")
            warn_nonzero(kflag, "kflag")
            warn_nonzero(vlen, "vlen")

        elif kind == BTF_KIND_DECL_TAG:
            (component_idx,) = unpack("i")
            print(
                f"[{idx}] {kind_name} '{name}' type_id={size} "
                + f"component_idx={component_idx}"
            )
            warn_nonzero(kflag, "kflag")
            warn_nonzero(vlen, "vlen")

        else:
            warn(
                f"<{idx}> Unexpected entry: kind={kind_name} "
                f"name_off={name_off} "
                f"vlen={vlen} kflag={kflag} size={size}"
            )

        idx += 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        warn(f"Usage: {sys.argv[0]} <btf_file>")
        sys.exit(1)
    print_btf(sys.argv[1])
