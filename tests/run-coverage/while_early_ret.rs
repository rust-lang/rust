#![allow(unused_assignments)]
// failure-status: 1

fn main() -> Result<(),u8> {
    let mut countdown = 10;
    while
        countdown
            >
        0
    {
        if
            countdown
                <
            5
        {
            return
                if
                    countdown
                        >
                    8
                {
                    Ok(())
                }
                else
                {
                    Err(1)
                }
                ;
        }
        countdown
            -=
        1
        ;
    }
    Ok(())
}

// ISSUE(77553): Originally, this test had `Err(1)` on line 22 (instead of `Ok(())`) and
// `std::process::exit(2)` on line 26 (instead of `Err(1)`); and this worked as expected on Linux
// and MacOS. But on Windows (MSVC, at least), the call to `std::process::exit()` exits the program
// without saving the InstrProf coverage counters. The use of `std::process:exit()` is not critical
// to the coverage test for early returns, but this is a limitation that should be fixed.
