#![deny(unused_variables)]

fn main() {
    let Some(_): Option<u32> = ({
        let x = 1; //~ ERROR unused variable: `x`
        Some(1)
    }) else {
        return;
    };

    #[allow(unused_variables)]
    let Some(_): Option<u32> = ({
        let x = 1;
        Some(1)
    }) else {
        return;
    };

    let Some(_): Option<u32> = ({
        #[allow(unused_variables)]
        let x = 1;
        Some(1)
    }) else {
        return;
    };

    let x = 1; //~ ERROR unused variable: `x`
}
