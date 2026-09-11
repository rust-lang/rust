fn issue_2995() {
    // '\u{2028}' is inserted in the code below.

    [0,  1];
    [0,  /* */ 1];
     [ 0 , 1 ] ;
}
