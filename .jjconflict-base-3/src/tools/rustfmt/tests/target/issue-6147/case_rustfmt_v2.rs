// rustfmt-style_edition: 2024

pub fn main() {
    let a = Some(12);
    match a {
        #![attr1]
        #![attr2]
        #![attr3]
        _ => None,
    }

    {
        match a {
            #![attr1]
            #![attr2]
            #![attr3]
            _ => None,
        }
    }
}
