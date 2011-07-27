./src/test/compile-fail/tail-non-call.rs:5:6:5:7: [1;31merror:[0m Non-call expression in tail call
./src/test/compile-fail/tail-non-call.rs:5   be x;
                                                 ^
rt: ---
rt: f00e:main:main:                   upcall fail 'explicit failure', src/comp/syntax/parse/parser.rs:112
rt: f00e:main:                        domain main @0x9d9404c root task failed
