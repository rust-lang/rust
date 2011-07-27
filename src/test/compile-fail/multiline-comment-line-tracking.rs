./src/test/compile-fail/multiline-comment-line-tracking.rs:9:2:9:3: [1;31merror:[0m unexpected token: %
./src/test/compile-fail/multiline-comment-line-tracking.rs:9   %; // parse error on line 9, but is reported on line 6 instead.
                                                               ^
rt: ---
rt: f00e:main:main:                   upcall fail 'explicit failure', src/comp/syntax/parse/parser.rs:112
rt: f00e:main:                        domain main @0x969e04c root task failed
