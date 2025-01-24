//@ build-pass
//@ edition:2018

#![feature(if_let_guard)]
// FIXME(static_mut_refs): Do not allow `static_mut_refs` lint
#![allow(static_mut_refs)]

static mut A: [i32; 5] = [1, 2, 3, 4, 5];

fn is_send_sync<T: Send + Sync>(_: T) {}

async fn fun() {
    let u = unsafe { A[async { 1 }.await] };
    unsafe {
        match A {
            i if async { true }.await => (),
            i if let Some(1) = async { Some(1) }.await => (),
            _ => (),
        }
    }
}

fn main() {
    let index_block = async {
        let u = unsafe { A[async { 1 }.await] };
    };
    let match_block = async {
        unsafe {
            match A {
                i if async { true }.await => (),
                i if let Some(2) = async { Some(2) }.await => (),
                _ => (),
            }
        }
    };
    is_send_sync(index_block);
    is_send_sync(match_block);
    is_send_sync(fun());
}
