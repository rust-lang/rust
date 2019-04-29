trait Tr {
    type TrAssoc;
}

struct Bar<'a, Item: Tr, <Item as Tr>::TrAssoc: 'a> {
//~^ ERROR qualified paths are not allowed in generic parameters
    item: Item,
    item_sub: &'a <Item as Tr>::TrAssoc,
}

fn main() {}
