import gdb


class PersonPrinter:
    "Print a Person"

    def __init__(self, val):
        self.val = val
        self.name = val["name"]
        self.age = int(val["age"])

    def to_string(self):
        return "{} is {} years old.".format(self.name, self.age)


def lookup(val):
    lookup_tag = val.type.tag
    if lookup_tag is None:
        return None
    if "dependency_with_embedded_visualizers::Person" == lookup_tag:
        return PersonPrinter(val)

    return None


gdb.current_objfile().pretty_printers.append(lookup)
