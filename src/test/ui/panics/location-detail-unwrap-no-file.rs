// run-fail
// check-run-results
// compile-flags: -Zlocation-detail=line,column

fn main() {
    let opt: Option<u32> = None;
    opt.unwrap();
}
