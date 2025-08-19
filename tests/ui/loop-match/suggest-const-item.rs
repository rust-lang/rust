#![allow(incomplete_features)]
#![feature(loop_match)]
#![feature(generic_const_items)]
#![crate_type = "lib"]

const fn const_fn() -> i32 {
    1
}

#[unsafe(no_mangle)]
fn suggest_const_block<const N: i32>() -> i32 {
    let mut state = 0;
    #[loop_match]
    loop {
        state = 'blk: {
            match state {
                0 => {
                    #[const_continue]
                    break 'blk const_fn();
                    //~^ ERROR could not determine the target branch for this `#[const_continue]`
                }
                1 => {
                    #[const_continue]
                    break 'blk const { const_fn() };
                    //~^ ERROR could not determine the target branch for this `#[const_continue]`
                }
                2 => {
                    #[const_continue]
                    break 'blk N;
                    //~^ ERROR could not determine the target branch for this `#[const_continue]`
                }
                _ => {
                    #[const_continue]
                    break 'blk 1 + 1;
                    //~^ ERROR could not determine the target branch for this `#[const_continue]`
                }
            }
        }
    }
    state
}

struct S;

impl S {
    const M: usize = 42;

    fn g() {
        let mut state = 0;
        #[loop_match]
        loop {
            state = 'blk: {
                match state {
                    0 => {
                        #[const_continue]
                        break 'blk Self::M;
                    }
                    _ => panic!(),
                }
            }
        }
    }
}

trait T {
    const N: usize;

    fn f() {
        let mut state = 0;
        #[loop_match]
        loop {
            state = 'blk: {
                match state {
                    0 => {
                        #[const_continue]
                        break 'blk Self::N;
                        //~^ ERROR could not determine the target branch for this `#[const_continue]`
                    }
                    _ => panic!(),
                }
            }
        }
    }
}

impl T for S {
    const N: usize = 1;
}

impl S {
    fn h() {
        let mut state = 0;
        #[loop_match]
        loop {
            state = 'blk: {
                match state {
                    0 => {
                        #[const_continue]
                        break 'blk Self::N;
                    }
                    _ => panic!(),
                }
            }
        }
    }
}

trait T2<U> {
    const L: u32;

    fn p() {
        let mut state = 0;
        #[loop_match]
        loop {
            state = 'blk: {
                match state {
                    0 => {
                        #[const_continue]
                        break 'blk Self::L;
                        //~^ ERROR could not determine the target branch for this `#[const_continue]`
                    }
                    _ => panic!(),
                }
            }
        }
    }
}

const SIZE_OF<T>: usize = size_of::<T>();

fn q<T>() {
    let mut state = 0;
    #[loop_match]
    loop {
        state = 'blk: {
            match state {
                0 => {
                    #[const_continue]
                    break 'blk SIZE_OF::<T>;
                    //~^ ERROR could not determine the target branch for this `#[const_continue]`
                }
                _ => panic!(),
            }
        }
    }
}

trait Trait<T> {
    const X: usize = 9000;
    const Y: usize = size_of::<T>();
}

impl<T> Trait<T> for () {}

fn r<T>() {
    let mut state = 0;
    #[loop_match]
    loop {
        state = 'blk: {
            match state {
                0 => {
                    #[const_continue]
                    break 'blk <() as Trait<T>>::X;
                }
                1 => {
                    #[const_continue]
                    break 'blk <() as Trait<T>>::Y;
                    //~^ ERROR could not determine the target branch for this `#[const_continue]`
                }
                _ => panic!(),
            }
        }
    }
}
