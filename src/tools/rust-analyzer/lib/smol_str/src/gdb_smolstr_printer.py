# Pretty printer for smol_str::SmolStr
#
# Usage (any of these):
#   (gdb) source /path/to/gdb_smolstr_printer.py
# or add to .gdbinit
#   python
#   import gdb
#   gdb.execute("source /path/to/gdb_smolstr_printer.py")
#   end
#
# After loading:
#   (gdb) info pretty-printer
#     ...
#     global pretty-printers:
#       smol_str
#         SmolStr
#
# Disable/enable:
#   (gdb) disable pretty-printer global smol_str SmolStr
#   (gdb) enable  pretty-printer global smol_str SmolStr

import gdb
import gdb.printing
import re

SMOL_INLINE_SIZE_RE = re.compile(r".*::_V(\d+)$")


def _read_utf8(mem):
    try:
        return mem.tobytes().decode("utf-8", errors="replace")
    except Exception:
        return repr(mem.tobytes())


def _active_variant(enum_val):
    """Return (variant_name, variant_value) for a Rust enum value using discriminant logic.
    Assume layout: fields[0] is unnamed u8 discriminant; fields[1] is the active variant.
    """
    fields = enum_val.type.fields()
    if len(fields) < 2:
        return None, None
    variant_field = fields[1]
    return variant_field.name, enum_val[variant_field]


class SmolStrProvider:
    def __init__(self, val):
        self.val = val

    def to_string(self):
        try:
            repr_enum = self.val["__0"]
        except Exception:
            return "<SmolStr: missing __0>"

        variant_name, variant_val = _active_variant(repr_enum)
        if not variant_name:
            return "<SmolStr: unknown variant>"

        if variant_name == "Inline":
            try:
                inline_len_val = variant_val["len"]
                m = SMOL_INLINE_SIZE_RE.match(str(inline_len_val))
                if not m:
                    return "<SmolStr Inline: bad len>"
                length = int(m.group(1))
                buf = variant_val["buf"]
                data = bytes(int(buf[i]) for i in range(length))
                return data.decode("utf-8", errors="replace")
            except Exception as e:
                return f"<SmolStr Inline error: {e}>"

        if variant_name == "Static":
            try:
                # variant_val["__0"] is &'static str
                return variant_val["__0"]
            except Exception as e:
                return f"<SmolStr Static error: {e}>"

        if variant_name == "Heap":
            try:
                # variant_val["__0"] is an Arc<str>
                inner = variant_val["__0"]["ptr"]["pointer"]
                # inner is a fat pointer to ArcInner<str>
                data_ptr = inner["data_ptr"]
                length = int(inner["length"])
                # ArcInner layout:
                # strong: Atomic<usize>, weak: Atomic<usize> | unsized tail 'data' bytes.
                sizeof_AtomicUsize = gdb.lookup_type(
                    "core::sync::atomic::AtomicUsize"
                ).sizeof
                header_size = sizeof_AtomicUsize * 2  # strong + weak counters
                data_arr = int(data_ptr) + header_size
                mem = gdb.selected_inferior().read_memory(data_arr, length)
                return _read_utf8(mem)
            except Exception as e:
                return f"<SmolStr Heap error: {e}>"

        return f"<SmolStr: unhandled variant {variant_name}>"

    def display_hint(self):
        return "string"


class SmolStrSubPrinter(gdb.printing.SubPrettyPrinter):
    def __init__(self):
        super(SmolStrSubPrinter, self).__init__("SmolStr")

    def __call__(self, val):
        if not self.enabled:
            return None
        try:
            t = val.type.strip_typedefs()
            if t.code == gdb.TYPE_CODE_STRUCT and t.name == "smol_str::SmolStr":
                return SmolStrProvider(val)
        except Exception:
            pass
        return None


class SmolStrPrettyPrinter(gdb.printing.PrettyPrinter):
    def __init__(self):
        super(SmolStrPrettyPrinter, self).__init__("smol_str", [])
        self.subprinters = []
        self._sp = SmolStrSubPrinter()
        self.subprinters.append(self._sp)

    def __call__(self, val):
        # Iterate subprinters (only one now, scalable for future)
        for sp in self.subprinters:
            pp = sp(val)
            if pp is not None:
                return pp
        return None


printer = SmolStrPrettyPrinter()


def register_printers(objfile=None):
    gdb.printing.register_pretty_printer(objfile, printer, replace=True)


register_printers()
