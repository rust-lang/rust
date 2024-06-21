fn bar() -> String {
    #[cfg(feature = )]
    [1, 2, 3].iter().map().collect::<String>() //~ ERROR expected `;`, found `#`

    #[attr] //~ ERROR attributes on expressions are experimental [E0658]
    //~^ ERROR cannot find attribute `attr`
    String::new()
}

fn main() { }
