#![feature(stmt_expr_attributes)]

fn foo() -> String {
    #[cfg(false)]
    [1, 2, 3].iter().map(|c| c.to_string()).collect::<String>() //~ ERROR expected `;`, found `#`
    #[cfg(not(FALSE))]
    String::new()
}

fn bar() -> String {
    #[attr]
    [1, 2, 3].iter().map(|c| c.to_string()).collect::<String>() //~ ERROR expected `;`, found `#`
    #[attr] //~ ERROR cannot find attribute `attr` in this scope
    String::new()
}

fn main() {
    println!("{}", foo());
}
