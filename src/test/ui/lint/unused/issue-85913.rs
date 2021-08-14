#![deny(unused_must_use)]

fn fire_missiles() -> bool {
    true
}

fn is_missiles_ready() -> bool {
    true
}

fn bang() -> ! {
    loop {}
}

pub fn fun() -> i32 {
    function() && return 1;
    is_missiles_ready() && fire_missiles();
    //~^ ERROR: unused logical operation that must be used
    !is_missiles_ready() || fire_missiles();
    //~^ ERROR: unused logical operation that must be used
    fire_missiles() || panic!("failed");
    fire_missiles() && bang();
    fire_missiles() || { panic!("failed"); };
    fire_missiles() && { bang(); };
    fire_missiles() || { panic!("failed") };
    fire_missiles() && { bang() };
    is_missiles_ready() && fire_missiles() || panic!("failed");
    !is_missiles_ready() || fire_missiles() && panic!("booom");
    //~^ ERROR: unused logical operation that must be used
    is_missiles_ready() && fire_missiles() || return 5;
    !is_missiles_ready() || {
        fire_missiles();
        panic!("booom");
    };
    !is_missiles_ready() || {
    //~^ ERROR: unused logical operation that must be used
        if fire_missiles() {
            panic!("booom");
        }
        println!("failed");
        false
    };
    fire_missiles() && fire_missiles() && panic!("2x booom");
    fire_missiles() && fire_missiles() || panic!("2x booom");
    return 0;
}

fn function() -> bool {
    true
}

fn main() {}
