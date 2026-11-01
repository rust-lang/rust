// rustfmt-max_width: 110
// rustfmt-use_small_heuristics: Max
// rustfmt-hard_tabs: true
// rustfmt-use_field_init_shorthand: true
// rustfmt-overflow_delimited_expr: true

// https://github.com/rust-lang/rustfmt/issues/4049
fn foo() {
	{
		{
			if let Some(MpcEv::PlayDrum(pitch, vel)) =
				// self.mpc.handle_input(e, /*btn_ctrl_down,*/ tx_launch_to_daw, state_view)
				self.mpc.handle_input(e, &mut MyBorrowedState { tx_launch_to_daw, state_view })
			{
				println!("bar");
			}

			if let Some(e) =
				// self.note_input.handle_input(e, /*btn_ctrl_down,*/ tx_launch_to_daw, state_view)
				self.note_input.handle_input(e, &mut MyBorrowedState { tx_launch_to_daw, state_view })
			{
				println!("baz");
			}
		}
	}
}
