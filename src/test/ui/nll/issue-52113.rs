trait Bazinga {}
impl<F> Bazinga for F {}

fn produce1<'a>(data: &'a u32) -> impl Bazinga + 'a {
    let x = move || {
        let _data: &'a u32 = data;
    };
    x
}

fn produce2<'a>(data: &'a mut Vec<&'a u32>, value: &'a u32) -> impl Bazinga + 'a {
    let x = move || {
        let value: &'a u32 = value;
        data.push(value);
    };
    x
}

fn produce3<'a, 'b: 'a>(data: &'a mut Vec<&'a u32>, value: &'b u32) -> impl Bazinga + 'a {
    let x = move || {
        let value: &'a u32 = value;
        data.push(value);
    };
    x
}

fn produce_err<'a, 'b: 'a>(data: &'b mut Vec<&'b u32>, value: &'a u32) -> impl Bazinga + 'b {
    let x = move || {
        let value: &'a u32 = value;
        data.push(value); //~ ERROR lifetime may not live long enough
    };
    x
}

fn main() {}
