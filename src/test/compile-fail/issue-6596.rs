macro_rules! e( //~ ERROR unknown macro variable `nonexistent`
    ($inp:ident) => (
        $nonexistent
    );
)

fn main() {
    e!(foo);
}
