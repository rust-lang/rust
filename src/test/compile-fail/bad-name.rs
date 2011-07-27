./src/test/compile-fail/bad-name.rs:4:7:4:8: [1;31merror:[0m expecting ident
./src/test/compile-fail/bad-name.rs:4   let x.y[int].z foo;
                                             ^
rt: ---
rt: f00e:main:main:                   upcall fail 'explicit failure', src/comp/syntax/parse/parser.rs:112
rt: f00e:main:                        domain main @0x8d6004c root task failed
