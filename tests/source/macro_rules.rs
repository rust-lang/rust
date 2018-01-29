macro_rules! m {
	// a
	($expr :expr,  $( $func : ident    ) *   ) => {
		{
		let    x =    $expr;
									                $func (
														x
											)
	}
	};

				/* b */

   	()           => {/* c */};

// d
( $item:ident  ) =>      {
	mod macro_item    {  struct $item ; }
};
}
