#[feature(macro_rules)];

// error-pattern: unknown macro variable `nonexistent`

macro_rules! e(
    ($inp:ident) => (
        $nonexistent
    );
)

fn main() {
    e!(foo);
}
