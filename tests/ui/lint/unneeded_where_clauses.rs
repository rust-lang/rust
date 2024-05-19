#[deny(unneeded_where_clauses)]

fn foo() where {} //~ ERROR this empty where clause is unneeded

fn main() {
    foo();
}
