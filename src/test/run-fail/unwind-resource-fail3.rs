// error-pattern:quux

struct faily_box {
    i: @int
}
// What happens to the box pointer owned by this class?
 
fn faily_box(i: @int) -> faily_box { faily_box { i: i } }

impl faily_box : Drop {
    fn finalize() {
        fail ~"quux";
    }
}

fn main() {
    faily_box(@10);
}
