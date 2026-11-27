fn main() {
    let greeting = "Hello, World!";
    let mut output = 0;
    for c in greeting.chars() {
        output += c;
        //~^ ERROR cannot add-assign `char` to `{integer}`
    }

    let mut sum: u32 = 0;
    sum += 'a';
    //~^ ERROR cannot add-assign `char` to `u32`

    let _ = 5 + '0';
    //~^ ERROR cannot add `char` to `{integer}`

    let _ = 'a' + 5;
    //~^ ERROR cannot add `{integer}` to `char`

    let mut small: u8 = 0;
    small += 'a';
    //~^ ERROR cannot add-assign `char` to `u8`

    let mut ch = 'a';
    ch += 5;
    //~^ ERROR binary assignment operation `+=` cannot be applied to type `char`
}
