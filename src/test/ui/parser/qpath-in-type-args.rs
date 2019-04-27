trait Tr {
    type TrSubtype;
}

struct Bar<'a, Item: Tr, <Item as Tr>::TrSubtype: 'a> {
//~^ ERROR qualified paths are not allowed in generic parameters
    item: Item,
    item_sub: &'a <Item as Tr>::TrSubtype,
}

fn main() {}
