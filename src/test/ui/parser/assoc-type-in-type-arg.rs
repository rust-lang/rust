trait Tr {
    type TrSubtype;
}

struct Bar<'a, Item: Tr, <Item as Tr>::TrSubtype: 'a> {
    //~^ ERROR associated type bounds do not belong here
    item: Item,
    item_sub: &'a <Item as Tr>::TrSubtype,
}

fn main() {}
