fn main() {
    let val = 42;
    let x = match val {
        (0 if true) => {
        //~^ ERROR expected identifier, found keyword `if`
        //~| ERROR expected one of `)`, `,`, `...`, `..=`, `..`, or `|`, found keyword `if`
        //~| ERROR expected one of `)`, `,`, `@`, or `|`, found keyword `true`
        //~| ERROR mismatched types
            42u8
        }
        _ => 0u8,
    };
    let _y: u32 = x; //~ ERROR mismatched types
}
