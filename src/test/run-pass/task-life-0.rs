// xfail-stage0
// xfail-stage1
// xfail-stage2
fn main() -> () {
    auto child_task = spawn child("Hello");
}

fn child(str s) {
    
}