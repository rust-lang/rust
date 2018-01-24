macro_rules! m {
	($expr :expr, $func : ident) => {
		{
		let    x =    $expr;
									                $func (
														x
											)
	}
	};

   	()           => {  };

( $item:ident  ) =>      {
	mod macro_item    {  struct $item ; }
};
}
