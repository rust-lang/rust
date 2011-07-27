./src/test/compile-fail/bad-alt.rs:5:7:5:8: [1;31merror:[0m expecting {, found ;
./src/test/compile-fail/bad-alt.rs:5   alt x;
                                            ^
rt: ---
rt: 0bb1:main:main:                   upcall fail 'explicit failure', src/comp/syntax/parse/parser.rs:112
rt: 0bb1:main:                        domain main @0x9e0f04c root task failed
