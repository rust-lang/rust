#![feature(core_intrinsics)]
//static TEST_FUNC_NAME: &'static [u8; 7] = b"main()\0";
  static TEST_FUNC_NAME: &'static [u8; 6] = b"main()";
fn main() {
    let a = 1;
    let b = 10;
    let _result = match {
        let _t = a < b;
        unsafe { core::intrinsics::instrprof_increment(TEST_FUNC_NAME as *const u8, 1234 as u64, 3 as u32, 0 as u32) };
        _t
    } {
        true => {
            let _t = true;
            unsafe { core::intrinsics::instrprof_increment(TEST_FUNC_NAME as *const u8, 1234 as u64, 3 as u32, 1 as u32) };
            _t
        }
        _ => false,
    };
}

/*

I NEED TO INSERT THE instrprof_increment() CALL:

  1. JUST BEFORE THE switchInt(_4) (because we haven't counted entering the function main() yet, deferring that to "JUST BEFORE FIRST BRANCH")
  2. SOME TIME AFTER THE switchInt(_4), AND JUST BEFORE ANOTHER BRANCH (in this case, before "goto")
  2.a. NOT BEFORE BOTH GOTO'S AFTER switchInt(_4) (because one can be calculated by counter expression), BUT PERHAPS INSERT A noop PLACEHOLDER
       AS A MARKER TO INCLUDE THE COVERAGE REGION AND REFERENCE THE COUNTERS TO BE SUBTRACTED (AND/OR SUMMED)?

    WHY DEFER INSERTING COUNTERS TO "JUST BEFORE FIRST BRANCH"? We can ignore panic/unwind() and only count if the coverage region ACTUALLY
    executed in entirety. BUT IS THAT NECESSARY? IS IT MUCH EASIER TO INSERT COUNTERS AT THE TOP OF A REGION THAT MUST EXECUTE IN ENTIRETY IF
    PANIC DOES NOT OCCUR? AND WHAT IF WE ADD SUPPORT FOR PANIC UNWIND (later)?

    IS THERE A BENEFIT OF THE DEFERRED APPROACH WHEN CONSIDERING EXPRESSIONS MAY HAVE EARLY RETURNS? (BECAUSE, WE STILL NEED TO COUNT THE REGION
    LEADING UP TO THE EXPRESSION ANYWAY)

=================================================
=================================================

To inject an intrinsic after computing a final expression value of a coverage region:

Replace the following basic block end (last statement plus terminator):

... <statements to compute _4> ...
StorageLive(_4)
StorageLive(_5)
_5 = _1
StorageLive(_6)
_6 = _2
_4 = Lt(move _5, move _6)
StorageDead(_6)
StorageDead(_5)
                             <------ to insert instrprof_increment() here
FakeRead(ForMatchedPlace, _4)
--------------------------------------------------------------------------------------
switchInt(_4)


=================================================
Insert call to intrinsic with:

StorageLive(_4)        # _4 is now meant for deferred FakeRead(ForMatchdPlace, _4) in BasicBlock after increment() call
StorageLive(_5)                 # Unchanged except _4 is now _5
StorageLive(_6)                 # Unchanged except _5 is now _6
_6 = _1                         # Unchanged except _5 is now _6
StorageLive(_7)                 # Unchanged except _6 is now _7
_7 = _2                         # Unchanged except _6 is now _7
_5 = Lt(move _6, move _7)       # Unchanged except _4, _5, _6 is now _5, _6, _7
StorageDead(_7)                 # Unchanged except _6 is now _7
StorageDead(_6)                 # Unchanged except _5 is now _6

FakeRead(ForLet, _5)   # CHANGED ForMatchedPlace to ForLet

> # ALL NEW AND NECESSARY TO CALL instrprof_increment()
> StorageLive(_8)        # ?? stores function pointer to instrprof_increment function?
> StorageLive(_9)
> StorageLive(_10)
> StorageLive(_11)
> _11 = const {alloc1+0: &&[u8; 6]}
> _10 = &raw const (*(*_11))
> _9 = move _10 as *const u8 (Pointer(ArrayToPointer))
> StorageDead(_10)
> StorageLive(_12)
> _12 = const 1234u64
> StorageLive(_13)
> _13 = const 3u32
> StorageLive(_14)
> _14 = const 0u32
> --------------------------------------------------------------------------------------
> _8 = const std::intrinsics::instrprof_increment(move _9, move _12, move _13, move _14)
>
> -> return
>
> StorageDead(_14)
> StorageDead(_13)
> StorageDead(_12)
> StorageDead(_9)
> StorageDead(_11)
> StorageDead(_8)

_4 = _5                         # ARE THESE LINES REDUNDANT? CAN I JUST PASS _5 DIRECTLY TO FakeRead()?
StorageDead(_5)                 # DROP "_t" temp result of `let _t = a < b`
                                # (NOTE THAT IF SO, I CAN REMOVE _5 altogether, and use _4, which coincidentally makes less changes)
                                # SEE BELOW

FakeRead(ForMatchedPlace, _4)   # Unchanged
--------------------------------------------------------------------------------------
switchInt(_4)                   # Unchanged


=================================================
Can I skip the extra variable and insert call to intrinsic with:

StorageLive(_4)             # Unchanged
StorageLive(_5)             # Unchanged
_5 = _1                     # Unchanged
StorageLive(_6)             # Unchanged
_6 = _2                     # Unchanged
_4 = Lt(move _5, move _6)   # Unchanged
StorageDead(_6)             # Unchanged
StorageDead(_5)             # Unchanged

> # ALL NEW AND NECESSARY TO CALL instrprof_increment()
> FakeRead(ForLet, _4)   # Save the post-increment result in temp "_t"
> StorageLive(_8)        # ?? stores function pointer to instrprof_increment function?
> StorageLive(_9)
> StorageLive(_10)
> StorageLive(_11)
> _11 = const {alloc1+0: &&[u8; 6]}
> _10 = &raw const (*(*_11))
> _9 = move _10 as *const u8 (Pointer(ArrayToPointer))
> StorageDead(_10)
> StorageLive(_12)
> _12 = const 1234u64
> StorageLive(_13)
> _13 = const 3u32
> StorageLive(_14)
> _14 = const 0u32
> --------------------------------------------------------------------------------------
> _8 = const std::intrinsics::instrprof_increment(move _9, move _12, move _13, move _14)
>
> -> return
>
> StorageDead(_14)
> StorageDead(_13)
> StorageDead(_12)
> StorageDead(_9)
> StorageDead(_11)
> StorageDead(_8)

FakeRead(ForMatchedPlace, _4)   # Unchanged  (PREVIOUSLY USED IN FakeRead(ForLet), is that OK?)
--------------------------------------------------------------------------------------
switchInt(_4)                   # Unchanged





=================================================
=================================================

For the second inserted call to instrprof_increment, without that call we have:

--------------------------------------------------------------------------------------
switchInt(_4)                   # From above

-> otherwise   # that is, "NOT false"

_3 = const true
                             <------ to insert instrprof_increment() here
--------------------------------------------------------------------------------------
goto

->                              # No label. No condition, and not a "return"

FakeRead(ForLet, _3)            # NOTE: Unused result
StorageDead(_4)
_0 = ()
StorageDead(_3)
StorageDead(_2)
StorageDead(_1)
--------------------------------------------------------------------------------------
goto

->                              # No label. No condition, and not a "return"

return   # from main()


=================================================
With the call to increment():

--------------------------------------------------------------------------------------
switchInt(_4)                   # From above

-> otherwise   # "NOT false"    # UNCHANGED

StorageLive(_15)                # CHANGED! Allocated new storage (_15) for the result of match, if true.
_15 = const true                # UNCHANGED except _3 is now _15
FakeRead(ForLet, _15)           # CHANGED! Assign value to temporary (to be assigned to _3 later) ... Do I need to do this?

> # ALL NEW AND NECESSARY TO CALL instrprof_increment()
> StorageLive(_16)                # pointer to instrprof_increment() function ?
> StorageLive(_17)
> StorageLive(_18)
> StorageLive(_19)
> _19 = const {alloc1+0: &&[u8; 6]}
> _18 = &raw const (*(*_19))
> _17 = move _18 as *const u8 (Pointer(ArrayToPointer))
> StorageDead(_18)
> StorageLive(_20)
> _20 = const 1234u64
> StorageLive(_21)
> _21 = const 3u32
> StorageLive(_22)
> _22 = const 1u32
> --------------------------------------------------------------------------------------
> _16 = const std::intrinsics::instrprof_increment(move _17, move _20, move _21, move _22)
>
> ->  return
>
> StorageDead(_22)
> StorageDead(_21)
> StorageDead(_20)
> StorageDead(_17)
> StorageDead(_19)
> StorageDead(_16)
> _3 = _15
> StorageDead(_15)

--------------------------------# UNCHANGED-------------------------------------------
goto                            # UNCHANGED

->                              # UNCHANGED

FakeRead(ForLet, _3)            # UNCHANGED
StorageDead(_4)                 # UNCHANGED
_0 = ()                         # UNCHANGED
StorageDead(_3)                 # UNCHANGED
StorageDead(_2)                 # UNCHANGED
StorageDead(_1)                 # UNCHANGED
--------------------------------------------------------------------------------------
goto                            # UNCHANGED

->                              # UNCHANGED

return   # from main()          # UNCHANGED

=================================================
As before, can I skip the extra variable (_15) and insert the call to intrinsic with _3 directly?:


--------------------------------------------------------------------------------------
switchInt(_4)                   # From above

-> otherwise   # "NOT false"    # UNCHANGED

_3 = const true                 # UNCHANGED?

> # ALL NEW AND NECESSARY TO CALL instrprof_increment()
> StorageLive(_16)                # pointer to instrprof_increment() function ?
> StorageLive(_17)
> StorageLive(_18)
> StorageLive(_19)
> _19 = const {alloc1+0: &&[u8; 6]}
> _18 = &raw const (*(*_19))
> _17 = move _18 as *const u8 (Pointer(ArrayToPointer))
> StorageDead(_18)
> StorageLive(_20)
> _20 = const 1234u64
> StorageLive(_21)
> _21 = const 3u32
> StorageLive(_22)
> _22 = const 1u32
> --------------------------------------------------------------------------------------
> _16 = const std::intrinsics::instrprof_increment(move _17, move _20, move _21, move _22)
>
> ->  return
>
> StorageDead(_22)
> StorageDead(_21)
> StorageDead(_20)
> StorageDead(_17)
> StorageDead(_19)
> StorageDead(_16)

--------------------------------# UNCHANGED-------------------------------------------
goto                            # UNCHANGED

->                              # UNCHANGED

FakeRead(ForLet, _3)            # UNCHANGED
StorageDead(_4)                 # UNCHANGED
_0 = ()                         # UNCHANGED
StorageDead(_3)                 # UNCHANGED
StorageDead(_2)                 # UNCHANGED
StorageDead(_1)                 # UNCHANGED
--------------------------------------------------------------------------------------
goto                            # UNCHANGED

->                              # UNCHANGED

return   # from main()          # UNCHANGED

*/