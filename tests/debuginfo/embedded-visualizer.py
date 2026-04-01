import gdb


class LinePrinter:
    "Print a Line"

    def __init__(self, val):
        self.val = val
        self.a = val["a"]
        self.b = val["b"]

    def to_string(self):
        return "({}, {})".format(self.a, self.b)


def lookup(val):
    lookup_tag = val.type.tag
    if lookup_tag is None:
        return None
    if "embedded_visualizer::Line" == lookup_tag:
        return LinePrinter(val)

    return None


gdb.current_objfile().pretty_printers.append(lookup)
