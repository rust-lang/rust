fn main() {
    1.create_a_type_error[ //~ `{integer}` is a primitive type and therefore doesn't have fields
        ()+() //~ ERROR binary operation `+` cannot be applied
              //   ^ ensure that we typeck the inner expression ^
    ];
}
