# Well-formedness checking

This chapter is mostly *to be written*. WF checking, in short, has the
job of checking that the various declarations in a Rust program are
well-formed. This is the basis for implied bounds, and partly for that
reason, this checking can be surprisingly subtle! (For example, we
have to be sure that each impl proves the WF conditions declared on
the trait.)



