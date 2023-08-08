// check-pass

pub enum UnOp {
    Not(Vec<()>),
}

pub fn foo() {
    if let Some(x) = None {
        match x {
            UnOp::Not(_) => {}
        }
    }
}

fn main() {
}
