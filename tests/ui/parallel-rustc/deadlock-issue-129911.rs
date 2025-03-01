// Test for #129911, which causes a deadlock bug
//
//@ parallel-front-end-robustness
//@ compile-flags: -Z threads=16

fn main() {
    type KooArc = Frc<
        {
            {
                {
                    {};
                }
                type Frc = Frc<{}>::Arc;;
            }
            type Frc = Frc<
                {
                    {
                        {
                            {};
                        }
                        type Frc = Frc<{}>::Arc;;
                    }
                    type Frc = Frc<
                        {
                            {
                                {
                                    {};
                                }
                                type Frc = Frc<{}>::Arc;;
                            }
                            type Frc = Frc<
                                {
                                    {
                                        {
                                            {};
                                        }
                                        type Frc = Frc<{}>::Arc;;
                                    }
                                    type Frc = Frc<
                                        {
                                            {
                                                {
                                                    {
                                                        {};
                                                    }
                                                    type Frc = Frc<{}>::Arc;;
                                                };
                                            }
                                            type Frc = Frc<
                                                {
                                                    {
                                                        {
                                                            {};
                                                        };
                                                    }
                                                    type Frc = Frc<{}>::Arc;;
                                                },
                                            >::Arc;;
                                        },
                                    >::Arc;;
                                },
                            >::Arc;;
                        },
                    >::Arc;;
                },
            >::Arc;;
        },
    >::Arc;
}
