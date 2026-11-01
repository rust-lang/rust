fn main() {
    let m = 42u32;

    let value = 'out: {
        match m {
            1 => break 'out Some(1u16),
            2 => Some(2u16),
            3 => break 'out Some(3u16),
            4 => break 'out Some(4u16),
            5 => break 'out Some(5u16),
            _ => {}
            //~^ ERROR  `match` arms have incompatible types
        }

        None
    };
}
