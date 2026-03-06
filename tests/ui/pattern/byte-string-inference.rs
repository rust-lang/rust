//@ check-pass
#![allow(todo_macro_uses)]

fn load<L>() -> Option<L> {
    todo!()
}

fn main() {
    while let Some(tag) = load() {
        match &tag {
            b"NAME" => {}
            b"DATA" => {}
            _ => {}
        }
    }
}
