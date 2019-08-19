// run-pass
// aux-build:issue-39823.rs

extern crate issue_39823;
use issue_39823::{RemoteC, RemoteG};

#[derive(Debug, PartialEq)]
struct LocalC(u32);

#[derive(Debug, PartialEq)]
struct LocalG<T>(T);

fn main() {
    let virtual_localc : &dyn Fn(_) -> LocalC = &LocalC;
    assert_eq!(virtual_localc(1), LocalC(1));

    let virtual_localg : &dyn Fn(_) -> LocalG<u32> = &LocalG;
    assert_eq!(virtual_localg(1), LocalG(1));

    let virtual_remotec : &dyn Fn(_) -> RemoteC = &RemoteC;
    assert_eq!(virtual_remotec(1), RemoteC(1));

    let virtual_remoteg : &dyn Fn(_) -> RemoteG<u32> = &RemoteG;
    assert_eq!(virtual_remoteg(1), RemoteG(1));
}
