// xfail-pretty
#[path = "mod-merge-hack-template.rs"]
#[merge = "mod-merge-hack-inst.rs"]
mod myint32;

fn main() {
    assert myint32::bits == 32;
    assert myint32::min(10, 20) == 10;
}