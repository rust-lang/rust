fn foo<T>() where T: Default -> {
//~^ ERROR expected type, found `{`
//~^^ ERROR return type should be specified after the function parameters
    0
}

fn main() {}
