/* FIXME #7302
fn foo(v: &const [uint]) -> ~[uint] {
    v.to_owned()
}
*/

fn bar(v: &mut [uint]) -> ~[uint] {
    v.to_owned()
}

fn bip(v: &[uint]) -> ~[uint] {
    v.to_owned()
}

pub fn main() {
    let mut the_vec = ~[1u, 2, 3, 100];
//    assert_eq!(the_vec.clone(), foo(the_vec));
    assert_eq!(the_vec.clone(), bar(the_vec));
    assert_eq!(the_vec.clone(), bip(the_vec));
}
