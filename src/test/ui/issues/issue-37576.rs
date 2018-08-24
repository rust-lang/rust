fn main() {
    'test_1: while break 'test_1 {}
    while break {}
    //~^ ERROR `break` or `continue` with no label

    'test_2: while let true = break 'test_2 {}
    while let true = break {}
    //~^ ERROR `break` or `continue` with no label

    loop { 'test_3: while break 'test_3 {} }
    loop { while break {} }
    //~^ ERROR `break` or `continue` with no label

    loop {
        'test_4: while break 'test_4 {}
        break;
    }
    loop {
        while break {}
        //~^ ERROR `break` or `continue` with no label
        break;
    }

    'test_5: while continue 'test_5 {}
    while continue {}
    //~^ ERROR `break` or `continue` with no label

    'test_6: while let true = continue 'test_6 {}
    while let true = continue {}
    //~^ ERROR `break` or `continue` with no label

    loop { 'test_7: while continue 'test_7 {} }
    loop { while continue {} }
    //~^ ERROR `break` or `continue` with no label

    loop {
        'test_8: while continue 'test_8 {}
        continue;
    }
    loop {
        while continue {}
        //~^ ERROR `break` or `continue` with no label
        continue;
    }
}
