// rustfmt-match_arm_indent: false

fn single_oneline() {
	match value { #[cfg(sslv2)] Sslv2 => handle(), }
}

fn single_multiline() {
	match value {
		Sslv3 => handle(),
		#[cfg(sslv2)] Sslv2 => { handle1(); handle2();}
		#[cfg(TLSv1)] Tlsv1 if condition || something_else || and_a_third_thing || long_condition || basically  => {}
		#[cfg(sslv23)] Sslv23 if condition || something_else || and_a_third_thing || long_condition || basically  => {actuall_content(); "ret";}
	}
}

fn multiple() {
	match value {
		Sslv3 => handle(),
		#[attr] #[cfg(sslv2)] Sslv2 => { handle1(); handle2();}
		#[attr] #[cfg(TLSv1)] Tlsv1 if condition || something_else || and_a_third_thing || long_condition || basically  => {}

		#[attr] #[cfg(sslv23)] Sslv23 if condition || something_else || and_a_third_thing || long_condition || basically  => {actuall_content(); "ret";}
	}
}
