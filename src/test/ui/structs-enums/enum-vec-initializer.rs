// run-pass
// pretty-expanded FIXME #23616

enum Flopsy {
    Bunny = 2
}

const BAR:usize = Flopsy::Bunny as usize;
const BAR2:usize = BAR;

pub fn main() {
    let _v = [0;  Flopsy::Bunny as usize];
    let _v = [0;  BAR];
    let _v = [0;  BAR2];
    const BAR3:usize = BAR2;
    let _v = [0;  BAR3];
}
