//@rustc-env: CLIPPY_PRINT_MIR=1

struct A {
    field: String,
}

struct Magic<'a> {
    a: &'a String,
}

const DUCK: u32 = 10;

#[forbid(clippy::borrow_pats)]
fn magic(input: &A, cond: bool) -> &str {
    if cond { "default" } else { &input.field }
}

impl A {
    // #[warn(clippy::borrow_pats)]
    fn borrow_self(&self) -> &A {
        self
    }

    // #[warn(clippy::borrow_pats)]
    fn borrow_field_direct(&self) -> &String {
        &self.field
    }

    // #[warn(clippy::borrow_pats)]
    fn borrow_field_deref(&self) -> &str {
        &self.field
    }

    fn borrow_field_or_default(&self) -> &str {
        if self.field.is_empty() {
            "Here be defaults"
        } else {
            &self.field
        }
    }

    fn borrow_field_into_mut_arg<'a>(&'a self, magic: &mut Magic<'a>) {
        magic.a = &self.field;
    }
}

fn main() {}
