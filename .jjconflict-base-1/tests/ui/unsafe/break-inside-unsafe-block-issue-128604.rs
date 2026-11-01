fn main() {
    let a = ["_"; unsafe { break; 1 + 2 }];
    //~^ ERROR `break` outside of a loop or labeled block

    unsafe {
        {
            //~^ HELP consider labeling this block to be able to break within it
            break;
            //~^ ERROR `break` outside of a loop or labeled block
        }
    }

    unsafe {
        break;
        //~^ ERROR `break` outside of a loop or labeled block
    }

    {
        //~^ HELP consider labeling this block to be able to break within it
        unsafe {
            break;
            //~^ ERROR `break` outside of a loop or labeled block
        }
    }

    while 2 > 1 {
        unsafe {
            if true || false {
                break;
            }
        }
    }

}
