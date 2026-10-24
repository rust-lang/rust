#![warn(clippy::for_unbounded_range)]

fn do_something<T: Copy>(_t: T) {}

fn main() {
    for i in 0_u8..=u8::MAX {
        do_something(i);
    }

    for i in 0_u8.. {
        if i > 2 {
            break;
        }
        do_something(i);
    }

    'outer: for i in 0_u8..10 {
        for i in 0_u8.. {
            if i > 2 {
                break 'outer;
            }
            do_something(i);
        }
    }

    for i in 0_u8.. {
        //~^ for_unbounded_range
        do_something(i);
    }

    for i in '\0'.. {
        //~^ for_unbounded_range
        do_something(i);
    }
}
