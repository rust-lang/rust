// pp-exact

fn main() {
    let x = match { 5 } { 1 => 5, 2 => 6, _ => 7, };
    assert_eq!(x , 7);
}
