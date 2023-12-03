fn bar() -> String {
    #[cfg]
    [1, 2, 3].iter() //~ ERROR expected `;`, found `#`
    #[feature]
    attr::fn bar() -> String { //~ ERROR expected identifier, found keyword `fn`
    //~^ ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `->`
    //~| ERROR expected `;`, found `bar`
    #[attr]
    [1, 2, 3].iter().map().collect::<String>()
    #[attr]

}()
}

fn main() { }
