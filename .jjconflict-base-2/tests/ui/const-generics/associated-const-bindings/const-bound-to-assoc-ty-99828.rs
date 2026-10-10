fn get_iter(vec: &[i32]) -> impl Iterator<Item = {}> + '_ {
    //~^ ERROR expected type, found constant
    //~| ERROR expected type, found constant
    //~| ERROR associated const equality is incomplete
    vec.iter()
}

fn main() {
    let vec = Vec::new();
    let mut iter = get_iter(&vec);
    iter.next();
}
