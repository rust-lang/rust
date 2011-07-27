./src/test/compile-fail/ext-noname.rs:4:3:4:4: [1;31merror:[0m expected a syntax expander name
./src/test/compile-fail/ext-noname.rs:4   #();
                                           ^
rt: ---
rt: 0bb1:main:main:                   upcall fail 'explicit failure', src/comp/syntax/parse/parser.rs:112
rt: 0bb1:main:                        domain main @0x91a204c root task failed
