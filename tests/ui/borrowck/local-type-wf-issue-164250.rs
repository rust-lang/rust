//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ compile-flags: -Zpolonius=off
//@[current] compile-flags: -Znext-solver=coherence
//@[next] compile-flags: -Znext-solver=globally

use std::sync::Mutex;

static KEPT: Mutex<Vec<Box<dyn std::any::Any + Send>>> = Mutex::new(Vec::new());

struct Q<T: 'static + Send>(Option<T>);

impl<T: 'static + Send> Drop for Q<T> {
    fn drop(&mut self) {
        if let Some(t) = self.0.take() {
            KEPT.lock().unwrap().push(Box::new(t));
        }
    }
}

fn implicit_drop() {
    let text = String::from("temporary");
    let mut value = Q(None);
    value.0 = Some(&text);
    //~^ ERROR `text` does not live long enough
}

fn explicit_drop() {
    let text = String::from("temporary");
    let mut value = Q(None);
    value.0 = Some(&text);
    //~^ ERROR `text` does not live long enough
    std::mem::drop(value);
}

fn main() {}
