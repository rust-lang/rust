fn main() {
    let _ = ['a'; { break 2; 1 }];
    //~^ ERROR `break` outside of a loop or labeled block
    //~| HELP consider labeling this block to be able to break within it

    const {
        {
            //~^ HELP consider labeling this block to be able to break within it
            break;
            //~^ ERROR `break` outside of a loop or labeled block
        }
    };

    const {
        break;
        //~^ ERROR `break` outside of a loop or labeled block
    };

    {
        const {
            break;
            //~^ ERROR `break` outside of a loop or labeled block
        }
    }
}

const FOO: () = break;
//~^ ERROR: `break` outside of a loop or labeled block

static BAR: () = break;
//~^ ERROR: `break` outside of a loop or labeled block
