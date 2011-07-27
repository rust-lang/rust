./src/test/compile-fail/self-missing-method.rs:6:14:6:15: [1;31merror:[0m expecting ., found (
./src/test/compile-fail/self-missing-method.rs:6           self();
                                                               ^
rt: ---
rt: 0bb1:main:main:                   upcall fail 'explicit failure', src/comp/syntax/parse/parser.rs:112
rt: 0bb1:main:                        domain main @0x8f5904c root task failed
