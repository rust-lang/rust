fn parse_num(num: String) -> i32 {
    num.parse()
    //~^ ERROR mismatched types [E0308]
}

fn main(){}
