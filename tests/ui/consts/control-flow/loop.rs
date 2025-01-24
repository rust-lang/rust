const _: () = loop { break (); };

static FOO: i32 = loop { break 4; };

const fn foo() {
    loop {}
}

pub trait Foo {
    const BAR: i32 = loop { break 4; };
}

impl Foo for () {
    const BAR: i32 = loop { break 4; };
}

fn non_const_outside() {
    const fn const_inside() {
        loop {}
    }
}

const fn const_outside() {
    fn non_const_inside() {
        loop {}
    }
}

fn main() {
    let x = [0; {
        while false {}
        4
    }];
}

const _: i32 = {
    let mut x = 0;

    while x < 4 {
        x += 1;
    }

    while x < 8 {
        x += 1;
    }

    x
};

const _: i32 = {
    let mut x = 0;

    for i in 0..4 {
        //~^ ERROR: cannot use `for`
        //~| ERROR: cannot use `for`
        x += i;
    }

    for i in 0..4 {
        //~^ ERROR: cannot use `for`
        //~| ERROR: cannot use `for`
        x += i;
    }

    x
};

const _: i32 = {
    let mut x = 0;

    loop {
        x += 1;
        if x == 4 {
            break;
        }
    }

    loop {
        x += 1;
        if x == 8 {
            break;
        }
    }

    x
};

const _: i32 = {
    let mut x = 0;
    while let None = Some(x) { }
    while let None = Some(x) { }
    x
};
