resource r(_r: int) {}

fn main() {
    let x = r(3);
    *x = 4; //! ERROR assigning to resource content
}